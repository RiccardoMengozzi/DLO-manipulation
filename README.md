# DLO-manipulation

modified `rigid_entity.py` at line 1237 inserting, to add forward kinematics.:

```
   @gs.assert_built
    def forward_kinematics(self, qpos, qs_idx_local=None, links_idx_local=None, envs_idx=None):
        """
        Compute forward kinematics for a single target link.

        Parameters
        ----------
        qpos : array_like, shape (n_qs,) or (n_envs, n_qs) or (len(envs_idx), n_qs)
            The joint positions.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Defaults to None.
        links_idx_local : None | array_like, optional
            The indices of the links to get. If None, all links will be returned. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        links_pos : array_like, shape (n_links, 3) or (n_envs, n_links, 3) or (len(envs_idx), n_links, 3)
            The positions of the links (link frame origins).
        links_quat : array_like, shape (n_links, 4) or (n_envs, n_links, 4) or (len(envs_idx), n_links, 4)
            The orientations of the links.
        """

        if self._solver.n_envs == 0:
            qpos = qpos.unsqueeze(0)
            envs_idx = torch.zeros(1, dtype=gs.tc_int)
        else:
            envs_idx = self._solver._sanitize_envs_idx(envs_idx)

        links_idx = torch.tensor(self._get_ls_idx(links_idx_local), dtype=torch.int32, device=gs.device)
        links_pos = torch.empty((len(envs_idx), len(links_idx), 3), dtype=gs.tc_float, device=gs.device)
        links_quat = torch.empty((len(envs_idx), len(links_idx), 4), dtype=gs.tc_float, device=gs.device)

        self._kernel_forward_kinematics(
            links_pos,
            links_quat,
            qpos,
            self._get_ls_idx(qs_idx_local),
            links_idx,
            envs_idx,
        )

        if self._solver.n_envs == 0:
            links_pos = links_pos.squeeze(0)
            links_quat = links_quat.squeeze(0)
        return links_pos, links_quat

    @ti.kernel
    def _kernel_forward_kinematics(
        self,
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        qpos: ti.types.ndarray(),
        qs_idx: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):

        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            # save original qpos
            # NOTE: reusing the IK_qpos_orig as cache (should not be a problem)
            self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]] = self._solver.qpos[qs_idx[i_q_], envs_idx[i_b_]]
            # set new qpos
            self._solver.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]
            # run FK
            self._solver._func_forward_kinematics_entity(self._idx_in_solver, envs_idx[i_b_])

        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                links_pos[i_b_, i_l_, i] = self._solver.links_state[links_idx[i_l_], envs_idx[i_b_]].pos[i]
            for i in ti.static(range(4)):
                links_quat[i_b_, i_l_, i] = self._solver.links_state[links_idx[i_l_], envs_idx[i_b_]].quat[i]

        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            # restore original qpos
            self._solver.qpos[qs_idx[i_q_], envs_idx[i_b_]] = self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]]
            # run FK
            self._solver._func_forward_kinematics_entity(self._idx_in_solver, envs_idx[i_b_])


```